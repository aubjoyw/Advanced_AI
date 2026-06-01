import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore", message="Not enough SMs")

"""
Diffusion Model with Transformer (DiT) for Images
===================================================
A clean, self-contained implementation of a Diffusion Transformer (DiT)
based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022).

Architecture overview:
  1. Patchify input image into a sequence of flattened patches
  2. Add sinusoidal positional embeddings
  3. Condition on timestep t and optional class label via adaLN-Zero
  4. N DiT blocks: adaLN-Zero → MHSA → adaLN-Zero → MLP (with residuals)
  5. Final LayerNorm + Linear head → unpatchify → predicted noise map

Usage:
  python dit_diffusion.py          # trains on random data (demo)
  python dit_diffusion.py --train  # same, explicit flag
"""

import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torchvision import datasets, transforms

# ─────────────────────────────────────────────
#  1.  Sinusoidal timestep embeddings
# ─────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Maps scalar timestep t ∈ [0, T] → R^d using sinusoidal frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)  →  out: (B, dim)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]      # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ─────────────────────────────────────────────
#  2.  adaLN-Zero conditioning module
# ─────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero.

    Projects the conditioning vector c into 6 scale/shift/gate scalars:
        γ₁, β₁  → modulate LayerNorm before attention
        α₁       → gate attention output
        γ₂, β₂  → modulate LayerNorm before MLP
        α₂       → gate MLP output

    All linear weights are zero-initialised so each DiT block starts
    as an identity function (critical for training stability).
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        # Zero-init: blocks start as identity → stable gradients
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        x: (B, N, D)  – patch tokens
        c: (B, D_c)   – conditioning vector

        Returns modulated x for attn, and modulated x for mlp,
        plus the two gate scalars α₁, α₂.
        """
        params = self.proj(c)                # (B, 6D)
        γ1, β1, α1, γ2, β2, α2 = params.chunk(6, dim=-1)
        # Unsqueeze for broadcast over sequence dim
        γ1, β1, α1 = γ1[:, None], β1[:, None], α1[:, None]
        γ2, β2, α2 = γ2[:, None], β2[:, None], α2[:, None]
        x_norm = self.norm(x)
        x_attn = (1 + γ1) * x_norm + β1   # modulated for attention
        x_mlp  = (1 + γ2) * x_norm + β2   # modulated for MLP
        return x_attn, x_mlp, α1, α2


# ─────────────────────────────────────────────
#  3.  Multi-head Self-Attention
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # each (B, N, H, d_h)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, N, d_h)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# ─────────────────────────────────────────────
#  4.  Point-wise Feed-Forward Network
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  5.  DiT Block
# ─────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    One DiT transformer block:

        x → adaLN(x, c) → MHSA → α₁·out + x
          → adaLN(x, c) → MLP  → α₂·out + x
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 cond_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.adaln = AdaLNZero(hidden_dim, cond_dim)
        self.attn  = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.mlp   = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_attn, x_mlp, α1, α2 = self.adaln(x, c)
        x = x + α1 * self.attn(x_attn)   # residual + gated attention
        x = x + α2 * self.mlp(x_mlp)     # residual + gated MLP
        return x


# ─────────────────────────────────────────────
#  6.  DiT (full model)
# ─────────────────────────────────────────────

class DiT(nn.Module):
    """
    Diffusion Transformer for images.

    Args:
        img_size:    H = W of square input image (pixels)
        patch_size:  p × p patch size (img_size must be divisible by p)
        in_channels: number of image channels (1 = grayscale, 3 = RGB)
        hidden_dim:  transformer hidden dimension D
        depth:       number of DiT blocks N
        num_heads:   attention heads (hidden_dim % num_heads == 0)
        mlp_ratio:   FFN expansion ratio
        num_classes: 0 = unconditional, >0 = class-conditional
        dropout:     dropout probability
    """

    def __init__(
        self,
        img_size:    int   = 32,
        patch_size:  int   = 4,
        in_channels: int   = 1,
        hidden_dim:  int   = 256,
        depth:       int   = 6,
        num_heads:   int   = 8,
        mlp_ratio:   float = 4.0,
        num_classes: int   = 0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim        = in_channels * patch_size * patch_size

        # ── Patch embedding ──
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        # ── Positional embedding (learnable) ──
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Timestep embedding ──
        t_dim = hidden_dim
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Class embedding (optional) ──
        self.class_embed = (
            nn.Embedding(num_classes + 1, hidden_dim)   # +1 for unconditional token
            if num_classes > 0 else None
        )

        cond_dim = hidden_dim

        # ── DiT blocks ──
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim, dropout)
            for _ in range(depth)
        ])

        # ── Final norm + output projection ──
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.proj_out = nn.Linear(hidden_dim, patch_dim)

        self._init_weights()

    def _init_weights(self):
        # Standard ViT-style init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Patchify / Unpatchify helpers ──

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        # Reshape into patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)          # (B, H/p, W/p, C, p, p)
        x = x.reshape(B, self.num_patches, -1)    # (B, N, C*p*p)
        return x

    def unpatchify(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)"""
        B, N, _ = x.shape
        p = self.patch_size
        C = self.in_channels
        h = w = img_size // p
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)          # (B, C, h, p, w, p)
        x = x.reshape(B, C, img_size, img_size)
        return x

    def forward(
        self,
        x:       torch.Tensor,          # (B, C, H, W) noisy image
        t:       torch.Tensor,          # (B,)          timestep indices
        labels:  torch.Tensor = None,   # (B,)          class labels (optional)
    ) -> torch.Tensor:
        """Returns predicted noise ε_θ(x_t, t, c) with same shape as x."""
        img_size = x.shape[-1]

        # 1. Patchify + embed
        tokens = self.patchify(x)                  # (B, N, patch_dim)
        tokens = self.patch_embed(tokens)           # (B, N, D)
        tokens = tokens + self.pos_embed            # add positional info

        # 2. Build conditioning vector c = t_emb [+ class_emb]
        c = self.t_embed(t)                         # (B, D)
        if self.class_embed is not None and labels is not None:
            c = c + self.class_embed(labels)        # additive class conditioning

        # 3. DiT blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # 4. Output head
        tokens = self.norm_out(tokens)
        tokens = self.proj_out(tokens)              # (B, N, patch_dim)

        # 5. Unpatchify → noise prediction
        return self.unpatchify(tokens, img_size)    # (B, C, H, W)


# ─────────────────────────────────────────────
#  7.  DDPM Noise Scheduler
# ─────────────────────────────────────────────

class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model scheduler.

    Implements:
      • Linear β schedule: β_t = β_start + t*(β_end - β_start)/T
      • Forward process: q(x_t|x_0) = N(√ᾱ_t·x_0, (1-ᾱ_t)·I)
      • Reverse step:   p_θ(x_{t-1}|x_t) using predicted noise ε_θ
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.T      = num_timesteps
        self.device = device

        # Linear schedule
        betas   = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas  = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏_{s=1}^t αs

        self.register("betas",     betas)
        self.register("alphas",    alphas)
        self.register("alpha_bar", alpha_bar)
        # Convenient pre-computed quantities
        self.register("sqrt_alpha_bar",        alpha_bar.sqrt())
        self.register("sqrt_one_minus_abar",   (1 - alpha_bar).sqrt())
        self.register("sqrt_recip_alpha",      alphas.rsqrt())
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.register("posterior_variance",    betas * (1 - alpha_bar_prev) / (1 - alpha_bar))

    def register(self, name, tensor):
        setattr(self, name, tensor)

    # ── Forward process (add noise) ──

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Sample x_t ~ q(x_t|x_0).
        Returns (x_t, noise) where noise is the Gaussian added.
        """
        noise = torch.randn_like(x0)
        s1 = self.sqrt_alpha_bar[t][:, None, None, None]
        s2 = self.sqrt_one_minus_abar[t][:, None, None, None]
        return s1 * x0 + s2 * noise, noise

    # ── Reverse step (denoise one step) ──

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t_idx: int,
             predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Compute x_{t-1} from x_t using predicted noise ε_θ.
        Follows the DDPM reverse process equation.
        """
        β   = self.betas[t_idx]
        α   = self.alphas[t_idx]
        ᾱ   = self.alpha_bar[t_idx]
        s1m = self.sqrt_one_minus_abar[t_idx]

        # Predicted x₀ from current x_t and noise estimate
        x0_pred = (x_t - s1m * predicted_noise) / ᾱ.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Posterior mean
        coef1   = (α.sqrt() * (1 - self.alpha_bar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0))) / (1 - ᾱ)
        coef2   = (self.alpha_bar[t_idx - 1].sqrt() if t_idx > 0 else torch.tensor(1.0)) * β / (1 - ᾱ)
        mean    = self.sqrt_recip_alpha[t_idx] * (x_t - β / s1m * predicted_noise)

        if t_idx == 0:
            return mean
        noise   = torch.randn_like(x_t)
        var     = self.posterior_variance[t_idx]
        return mean + var.sqrt() * noise

    # ── Full sampling loop ──

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple,
               labels: torch.Tensor = None,
               cfg_scale: float = 1.0) -> torch.Tensor:
        """
        Generate images by iterating the reverse process from t=T to t=0.

        Args:
            model:     DiT model (in eval mode)
            shape:     (B, C, H, W) shape of images to generate
            labels:    optional class labels for conditional generation
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
        """
        device = self.device
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            if cfg_scale > 1.0 and labels is not None:
                # CFG: blend conditional & unconditional predictions
                uncond = torch.zeros_like(labels)
                eps_cond   = model(x, t_batch, labels)
                eps_uncond = model(x, t_batch, uncond)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_batch, labels)

            x = self.step(x, t, eps)

        return x


# ─────────────────────────────────────────────
#  8.  Training loop
# ─────────────────────────────────────────────

def train(
    img_size:      int   = 32,
    patch_size:    int   = 4,
    in_channels:   int   = 1,
    hidden_dim:    int   = 256,
    depth:         int   = 6,
    num_heads:     int   = 8,
    num_classes:   int   = 10,
    num_timesteps: int   = 1000,
    batch_size:    int   = 64,
    epochs:        int   = 5,
    lr:            float = 1e-4,
    device:        str   = "cpu",
):
    print(f"\n{'─'*55}")
    print("  Diffusion Transformer (DiT) — Training Demo")
    print(f"{'─'*55}")
    print(f"  Device      : {device}")
    print(f"  Image size  : {img_size}×{img_size}×{in_channels}")
    print(f"  Patch size  : {patch_size}×{patch_size}  ({(img_size//patch_size)**2} patches)")
    print(f"  Hidden dim  : {hidden_dim}  |  Depth: {depth}  |  Heads: {num_heads}")
    print(f"  Classes     : {num_classes}  |  Timesteps: {num_timesteps}")
    print(f"{'─'*55}\n")

    # ── Model + scheduler ──
    model = DiT(
        img_size=img_size, patch_size=patch_size, in_channels=in_channels,
        hidden_dim=hidden_dim, depth=depth, num_heads=num_heads,
        num_classes=num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {total_params:,}")

    # ── torch.compile (PyTorch 2.0+, ~20-30% speedup) ──
    use_compile = hasattr(torch, "compile") and device == "cuda"
    if use_compile:
        print("  torch.compile  : enabled")
        model = torch.compile(model)
    else:
        print("  torch.compile  : skipped (requires PyTorch 2.0+ and CUDA)")

    # ── Mixed precision scaler (fp16, ~30-50% speedup on RTX cards) ──
    use_amp = device == "cuda"
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"  Mixed precision: {'enabled (fp16)' if use_amp else 'disabled (CPU)'}\n")

    scheduler  = DDPMScheduler(num_timesteps, device=device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # ── Dataset ──
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # resize all to your target size
        transforms.RandomHorizontalFlip(),  # cheap augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],  # → [-1, 1] per channel
                             [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(root="data/Dogs", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    # ── Training ──
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x0, labels in loader:
            x0, labels = x0.to(device), labels.to(device)
            B = x0.size(0)

            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (B,), device=device)

            # Forward process: add noise
            x_t, noise = scheduler.add_noise(x0, t)

            # Classifier-free guidance: randomly drop class labels (10%)
            drop_mask = torch.rand(B, device=device) < 0.1
            labels_in = labels.clone()
            labels_in[drop_mask] = 0           # 0 = unconditional token

            # Predict noise (fp16 where possible via autocast)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                noise_pred = model(x_t, t, labels_in)
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * B

        lr_sched.step()
        avg_loss = epoch_loss / len(dataset)
        print(f"  Epoch {epoch:>3}/{epochs}  |  loss: {avg_loss:.5f}  |  lr: {lr_sched.get_last_lr()[0]:.2e}")

    print(f"\n{'─'*55}")
    print("  Training complete.")

    # ── Sample a batch ──
    n_samples = 8
    print(f"  Sampling {n_samples} images …")
    torch.set_float32_matmul_precision('high')
    model.eval()
    sample_labels = torch.randint(1, num_classes + 1, (n_samples,), device=device)
    samples = scheduler.sample(
        model, shape=(n_samples, in_channels, img_size, img_size),
        labels=sample_labels, cfg_scale=3.0,
    )
    print(f"  Generated tensor shape : {samples.shape}")
    print(f"  Value range            : [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"{'─'*55}\n")

    display_samples(samples, sample_labels, in_channels, save_path="dit_samples.png")

    return model, samples


# ─────────────────────────────────────────────
#  9.  Visualisation
# ─────────────────────────────────────────────

def display_samples(
    samples: torch.Tensor,       # (B, C, H, W)  raw model output
    labels:  torch.Tensor,       # (B,)           class indices
    in_channels: int = 1,
    save_path: str = "dit_samples.png",
):
    """
    Normalise and display the sampled images in a clean grid.
    Saves to *save_path* and shows the window if a display is available.
    """
    B = samples.shape[0]
    cols = min(B, 4)
    rows = math.ceil(B / cols)

    # Undo training normalisation: [-1, 1] → [0, 1]
    imgs = samples.detach().cpu().float()
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

    fig = plt.figure(figsize=(cols * 2.4, rows * 2.6 + 0.6), facecolor="#111111")
    fig.suptitle("DiT — generated samples", color="white", fontsize=13, y=0.98)

    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.15, wspace=0.08)

    for idx in range(B):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, c])

        img = imgs[idx]                           # (C, H, W)
        if in_channels == 1:
            ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))

        ax.set_title(f"class {labels[idx].item()}", color="#aaaaaa",
                     fontsize=9, pad=3)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Hide any unused subplots
    for idx in range(B, rows * cols):
        r, c = divmod(idx, cols)
        fig.add_subplot(gs[r, c]).axis("off")

    plt.savefig(save_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Samples saved → {save_path}")


# ─────────────────────────────────────────────
#  9.  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT Diffusion Model Demo")
    parser.add_argument("--img-size",      type=int,   default=32)
    parser.add_argument("--patch-size",    type=int,   default=4)
    parser.add_argument("--channels",      type=int,   default=1)
    parser.add_argument("--hidden-dim",    type=int,   default=256)
    parser.add_argument("--depth",         type=int,   default=6)
    parser.add_argument("--num-heads",     type=int,   default=8)
    parser.add_argument("--num-classes",   type=int,   default=10)
    parser.add_argument("--timesteps",     type=int,   default=1000)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--train",         action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, samples = train(
        img_size      = args.img_size,
        patch_size    = args.patch_size,
        in_channels   = args.channels,
        hidden_dim    = args.hidden_dim,
        depth         = args.depth,
        num_heads     = args.num_heads,
        num_classes   = args.num_classes,
        num_timesteps = args.timesteps,
        batch_size    = args.batch_size,
        epochs        = args.epochs,
        lr            = args.lr,
        device        = device,
    )