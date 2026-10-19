"""
models.py - model definitions for Part 4.

Currently contains:
  * Encoder / Decoder building blocks (shared by the discrete and continuous AE)
  * VectorQuantizerEMA
  * VQVAE   (discrete AE: 8x8 grid of codes, 1024 codes = 10 bits per position)
  * FSQAE   (discrete AE: 8x8x10 tensor, every entry quantised to `levels` values)  <- used from now on
  * ContinuousAE (continuous AE: 8x8x5 floats = 320 values)
  * MaskedTokenDenoiser   (step 04: discrete latent diffusion, absorbing/mask state, MaskGIT-style sampling)
  * ARLatentTransformer   (step 05: causal transformer + per-token diffusion head on continuous latents)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """32x32x3 -> 8x8xout_dim  (two stride-2 downsamplings)."""

    def __init__(self, hidden=128, out_dim=64, num_res=2):
        super().__init__()
        layers = [
            nn.Conv2d(3, hidden // 2, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        ]
        layers += [ResBlock(hidden) for _ in range(num_res)]
        layers += [nn.ReLU(), nn.Conv2d(hidden, out_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """8x8xin_dim -> 32x32x3."""

    def __init__(self, hidden=128, in_dim=64, num_res=2):
        super().__init__()
        layers = [nn.Conv2d(in_dim, hidden, 3, padding=1)]
        layers += [ResBlock(hidden) for _ in range(num_res)]
        layers += [
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden // 2, 3, 4, stride=2, padding=1),  # 32x32
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ----------------------------------------------------------------------------
# Vector quantizer with EMA codebook updates + dead-code restarts
# ----------------------------------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes=1024,
        dim=64,
        decay=0.99,
        eps=1e-5,
        commit_cost=0.25,
        restart_threshold=0.5,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.commit_cost = commit_cost
        self.restart_threshold = restart_threshold

        self.register_buffer("embed", torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("initted", torch.tensor(0, dtype=torch.uint8))

    def _lookup(self, flat):
        d = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1).unsqueeze(0)
        )
        return d.argmin(dim=1)

    def forward(self, z):
        # z: (B, C, H, W) -> (B*H*W, C)
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, C).float()

        # Initialise codebook from the first batch of encoder outputs
        if self.training and self.initted.item() == 0:
            idx = torch.randint(0, flat.size(0), (self.num_codes,), device=flat.device)
            self.embed.copy_(flat[idx].detach())
            self.embed_avg.copy_(self.embed)
            self.cluster_size.fill_(1.0)
            self.initted.fill_(1)

        idx = self._lookup(flat)
        onehot = F.one_hot(idx, self.num_codes).type(flat.dtype)
        quantized = F.embedding(idx, self.embed).view(B, H, W, C)

        if self.training:
            with torch.no_grad():
                counts = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                embed_sum = onehot.t() @ flat.detach()
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cs = (
                    (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
                )
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))

                # Restart codes that are (almost) never used -> prevents codebook collapse
                dead = self.cluster_size < self.restart_threshold
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    new = flat[
                        torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
                    ]
                    self.embed[dead] = new.detach()
                    self.embed_avg[dead] = new.detach()
                    self.cluster_size[dead] = 1.0

        commit_loss = self.commit_cost * F.mse_loss(z_perm, quantized.detach())
        quantized = z_perm + (quantized - z_perm).detach()  # straight-through
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        probs = onehot.mean(0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return quantized, commit_loss, idx.view(B, H, W), perplexity

    def codes_to_latent(self, idx):
        """idx: (B, H, W) long -> (B, C, H, W) float"""
        return F.embedding(idx, self.embed).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------------
# VQ-VAE
# ----------------------------------------------------------------------------
class VQVAE(nn.Module):
    """
    Discrete AE.  Latent = 8x8 grid of integers in [0, num_codes).
    With num_codes = 1024 -> 10 bits per position -> 8x8x10 bits total.
    """

    def __init__(self, hidden=128, num_codes=1024, code_dim=64, num_res=2):
        super().__init__()
        self.config = dict(
            hidden=hidden, num_codes=num_codes, code_dim=code_dim, num_res=num_res
        )
        self.vocab, self.n_sub = num_codes, 1
        self.encoder = Encoder(hidden, code_dim, num_res)
        self.quantizer = VectorQuantizerEMA(num_codes, code_dim)
        self.decoder = Decoder(hidden, code_dim, num_res)

    def forward(self, x):
        z = self.encoder(x)
        zq, vq_loss, idx, ppl = self.quantizer(z)
        recon = self.decoder(zq)
        return recon, vq_loss, idx, ppl

    @torch.no_grad()
    def encode(self, x):
        """x in [-1,1] -> code indices (B, 8, 8) long"""
        _, _, idx, _ = self.quantizer(self.encoder(x))
        return idx

    @torch.no_grad()
    def decode_codes(self, idx):
        """code indices (B, 8, 8) long -> images in [-1,1]"""
        return self.decoder(self.quantizer.codes_to_latent(idx))

    @torch.no_grad()
    def decode_tokens(self, tokens):
        """tokens (B, 64, 1) long -> images (same interface as FSQAE)"""
        return self.decode_codes(tokens.reshape(-1, 8, 8))


# ----------------------------------------------------------------------------
# Continuous autoencoder
# ----------------------------------------------------------------------------
class ContinuousAE(nn.Module):
    """
    Continuous AE.  Latent = (latent_ch, 8, 8) floats.
    latent_ch = 5 -> 8x8x5 = 320 values (assignment limit for continuous AEs).
    Reuses the same Encoder / Decoder as the VQ-VAE, just without the quantizer.
    """

    def __init__(self, hidden=128, latent_ch=5, num_res=2):
        super().__init__()
        self.config = dict(hidden=hidden, latent_ch=latent_ch, num_res=num_res)
        self.encoder = Encoder(hidden, latent_ch, num_res)
        self.decoder = Decoder(hidden, latent_ch, num_res)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, x):
        """x in [-1,1] -> latent (B, latent_ch, 8, 8) float"""
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z):
        """latent (B, latent_ch, 8, 8) -> images in [-1,1]"""
        return self.decoder(z)


# ----------------------------------------------------------------------------
# FSQ autoencoder (discrete AE with an 8x8x10 latent tensor, every entry quantised to `levels` values)
# ----------------------------------------------------------------------------
class FSQ(nn.Module):
    """Finite scalar quantisation: bound each entry with tanh, round it to one of `levels` values (STE)."""

    def __init__(self, levels=8):
        super().__init__()
        self.L = levels
        self.half = levels // 2

    def _bound(self, z, eps=1e-3):
        half_l = (self.L - 1) * (1 + eps) / 2
        offset = 0.5 if self.L % 2 == 0 else 0.0
        shift = math.atanh(offset / half_l) if offset else 0.0
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z):
        b = self._bound(z)
        r = torch.round(b)
        q = b + (r - b).detach()  # straight-through
        idx = (r.detach() + self.half).long()  # 0 .. L-1
        return q / self.half, idx

    def indices_to_latent(self, idx):
        return (idx.float() - self.half) / self.half


class FSQAE(nn.Module):
    """
    Discrete AE.  Latent = (channels, 8, 8) tensor, every entry an integer in [0, levels).
    channels = 10 -> latent tensor 8x8x10 (assignment limit).  levels=2 gives a pure binary 640-bit latent.
    """

    def __init__(self, hidden=256, channels=10, levels=8, num_res=2):
        super().__init__()
        self.config = dict(
            hidden=hidden, channels=channels, levels=levels, num_res=num_res
        )
        self.vocab, self.n_sub = levels, channels
        self.encoder = Encoder(hidden, channels, num_res)
        self.fsq = FSQ(levels)
        self.decoder = Decoder(hidden, channels, num_res)

    @property
    def bits_per_position(self):
        return self.n_sub * math.log2(self.vocab)

    def forward(self, x):
        zq, idx = self.fsq(self.encoder(x))  # idx (B, C, 8, 8)
        recon = self.decoder(zq)
        idx = idx.permute(0, 2, 3, 1).contiguous()  # (B, 8, 8, C)
        with torch.no_grad():  # mean per-channel perplexity (diagnostic only)
            ppl = 0.0
            for c in range(self.n_sub):
                p = torch.bincount(idx[..., c].flatten(), minlength=self.vocab).float()
                p = p / p.sum()
                ppl = ppl + torch.exp(-(p * torch.log(p + 1e-10)).sum())
            ppl = ppl / self.n_sub
        return recon, zq.new_zeros(()), idx, ppl

    @torch.no_grad()
    def encode(self, x):
        """x in [-1,1] -> indices (B, 8, 8, C) long"""
        return self.forward(x)[2]

    @torch.no_grad()
    def decode_codes(self, idx):
        """indices (B, 8, 8, C) long -> images in [-1,1]"""
        return self.decoder(self.fsq.indices_to_latent(idx.permute(0, 3, 1, 2)))

    @torch.no_grad()
    def decode_tokens(self, tokens):
        """tokens (B, 64, C) long -> images"""
        return self.decode_codes(tokens.reshape(-1, 8, 8, self.n_sub))


# ============================================================================
# STEP 04 - Discrete latent diffusion (absorbing-state / masked diffusion)
# ============================================================================
class MaskedTokenDenoiser(nn.Module):
    """
    Bidirectional transformer denoiser for discrete latents.
    Data layout     : (B, L, S) = L grid positions (64) x S sub-tokens per position (10 for FSQAE, 1 for VQVAE),
                      every entry an integer in [0, vocab).  vocab == [MASK].
    Forward process : every ENTRY is independently replaced by [MASK] (the absorbing state).
    Denoiser        : tokens (B, L, S) -> logits (B, L, S, vocab).  The S embeddings of a position are summed.
    Reverse process : start from all-[MASK] and unmask the most confident entries step by step
                      (MaskGIT-style iterative decoding with a cosine schedule).
    """

    def __init__(
        self,
        vocab=1024,
        seq_len=64,
        d_model=512,
        nhead=8,
        num_layers=8,
        ff_mult=4,
        dropout=0.1,
        n_sub=1,
    ):
        super().__init__()
        self.config = dict(
            vocab=vocab,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_mult=ff_mult,
            dropout=dropout,
            n_sub=n_sub,
        )
        self.vocab, self.seq_len, self.n_sub, self.mask_id = (
            vocab,
            seq_len,
            n_sub,
            vocab,
        )
        self.tok_emb = nn.Embedding(n_sub * (vocab + 1), d_model)
        self.register_buffer(
            "sub_offset", torch.arange(n_sub) * (vocab + 1), persistent=False
        )
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_model * ff_mult,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_sub * vocab)

    def forward(self, tokens):
        B, L, S = tokens.shape
        x = self.tok_emb(tokens + self.sub_offset).sum(dim=2) + self.pos_emb
        x = self.out_norm(self.encoder(x))
        return self.head(x).view(B, L, S, self.vocab)

    @torch.no_grad()
    def sample(self, n, steps=32, temperature=1.0, choice_temperature=4.5, device=None):
        """Returns (n, L, S) long tokens in [0, vocab). Call on a model in eval() mode."""
        device = device or next(self.parameters()).device
        L, S, V = self.seq_len, self.n_sub, self.vocab
        N = L * S
        tokens = torch.full((n, L, S), self.mask_id, dtype=torch.long, device=device)
        for s in range(steps):
            probs = torch.softmax(self(tokens).float() / temperature, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, V), 1).view(n, L, S)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            is_mask = tokens == self.mask_id
            tokens = torch.where(is_mask, sampled, tokens)

            ratio = (s + 1) / steps
            n_remask = int(N * math.cos(math.pi / 2 * ratio))  # 0 on the last step
            if n_remask > 0:
                conf, is_mask_f = conf.reshape(n, N), is_mask.reshape(n, N)
                gumbel = -torch.log(
                    -torch.log(torch.rand_like(conf).clamp(1e-10, 1 - 1e-10))
                )
                score = (
                    torch.log(conf.clamp_min(1e-10))
                    + choice_temperature * (1 - ratio) * gumbel
                )
                score = torch.where(
                    is_mask_f, score, torch.full_like(score, float("inf"))
                )  # keep fixed entries
                idx = score.topk(n_remask, dim=1, largest=False).indices
                tokens.view(n, N).scatter_(1, idx, self.mask_id)
        return tokens


# ============================================================================
# STEP 05 - Continuous latent transformer (causal AR + per-token diffusion head)
# ============================================================================
def timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def cosine_alphas_cumprod(T=1000, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class AdaLNBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.norm = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(w, 3 * w))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, h, c):
        shift, scale, gate = self.mod(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class DiffusionHead(nn.Module):
    """Small MLP that predicts the noise on ONE latent vector, given the transformer's output as condition."""

    def __init__(self, in_ch=5, cond_dim=512, width=512, depth=4):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, width)
        self.cond_proj = nn.Linear(cond_dim, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([AdaLNBlock(width) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, in_ch)
        for m in (self.final_mod[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, t, cond):
        c = self.cond_proj(cond) + self.time_mlp(timestep_embedding(t))
        h = self.in_proj(x_t)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        return self.out(self.final_norm(h) * (1 + scale) + shift)


class ARLatentTransformer(nn.Module):
    """
    Raster-order autoregressive model over the 64 positions of the 8x8 latent grid.
    Each position is a continuous `latent_ch`-vector.  A causal transformer reads the previous
    vectors and produces a condition h_i; a small diffusion head then models p(x_i | h_i)
    (no MSE regression -> no blurry averaging).
    """

    def __init__(
        self,
        latent_ch=5,
        seq_len=64,
        d_model=512,
        nhead=8,
        num_layers=8,
        dropout=0.1,
        head_width=512,
        head_depth=4,
        T=1000,
    ):
        super().__init__()
        self.config = dict(
            latent_ch=latent_ch,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            head_width=head_width,
            head_depth=head_depth,
            T=T,
        )
        self.latent_ch, self.seq_len, self.T = latent_ch, seq_len, T
        self.in_proj = nn.Linear(latent_ch, d_model)
        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_model * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = DiffusionHead(latent_ch, d_model, head_width, head_depth)
        self.register_buffer(
            "alphas_cumprod", cosine_alphas_cumprod(T), persistent=False
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1),
            persistent=False,
        )

    def condition(self, x):
        """x (B, L, C) -> h (B, L, d). h[:, i] only depends on x[:, :i]."""
        B = x.size(0)
        inp = (
            torch.cat([self.bos.expand(B, -1, -1), self.in_proj(x[:, :-1])], dim=1)
            + self.pos_emb
        )
        h = self.encoder(inp, mask=self.causal_mask, is_causal=True)
        return self.out_norm(h)

    def loss(self, x, t_repeat=4):
        """Noise-prediction loss. Each token's condition is reused with `t_repeat` different (t, noise) draws."""
        B, L, C = x.shape
        h = self.condition(x).reshape(B * L, -1).repeat(t_repeat, 1)
        x0 = x.reshape(B * L, C).repeat(t_repeat, 1)
        t = torch.randint(0, self.T, (x0.size(0),), device=x.device)
        noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].unsqueeze(1)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        pred = self.head(x_t, t, h)
        return F.mse_loss(pred.float(), noise)

    @torch.no_grad()
    def sample(self, n, steps=50, eta=1.0, temperature=1.0, x0_clip=6.0, device=None):
        """Returns (n, seq_len, latent_ch) NORMALISED latents. Call on a model in eval() mode.
        eta=1 -> DDPM-like stochastic sampling, eta=0 -> deterministic DDIM."""
        device = device or next(self.parameters()).device
        L, C = self.seq_len, self.latent_ch
        x = torch.zeros(n, L, C, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        one = torch.tensor(1.0, device=device)
        for i in range(L):
            h = self.condition(x)[:, i]
            y = torch.randn(n, C, device=device) * temperature
            for k in range(steps):
                t = ts[k]
                a_t = self.alphas_cumprod[t]
                last = k + 1 == steps
                a_prev = one if last else self.alphas_cumprod[ts[k + 1]]
                eps = self.head(y, t.expand(n), h).float()
                x0 = ((y - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(
                    -x0_clip, x0_clip
                )
                eps = (y - a_t.sqrt() * x0) / (1 - a_t).sqrt()
                sigma = (
                    eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
                )
                direction = (1 - a_prev - sigma**2).clamp_min(0).sqrt() * eps
                noise = (
                    torch.zeros_like(y) if last else torch.randn_like(y) * temperature
                )
                y = a_prev.sqrt() * x0 + direction + sigma * noise
            x[:, i] = y
        return x
