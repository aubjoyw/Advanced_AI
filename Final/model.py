import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes the timestep t as a frequency embedding (like in Transformers)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """
    Conv block with time AND optional conditioning embedding injection.
    Both embeddings are projected to out_ch and added after the first conv.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        # Conditioning projection — only created if conditioning is used
        self.cond_mlp = nn.Linear(cond_emb_dim, out_ch) if cond_emb_dim else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        # Add conditioning embedding if provided
        if self.cond_mlp is not None and c_emb is not None:
            h = h + self.cond_mlp(c_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention for capturing long-range spatial dependencies."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)                # (B, C, H, W)
        return x + h


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_dim=32, time_emb_dim=128,
                 num_classes=None, cond_emb_dim=128):
        """
        Args:
            img_channels:  Input/output image channels (3 for RGB).
            base_dim:      Base channel count. Doubles at each encoder level.
            time_emb_dim:  Dimension of the timestep embedding.
            num_classes:   Number of classes for conditioning. None = unconditional.
            cond_emb_dim:  Dimension of the class conditioning embedding.
        """
        super().__init__()
        self.num_classes = num_classes

        # --- Timestep embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Class conditioning embedding ---
        # num_classes + 1 to include a null/unconditional token at index num_classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + 1, cond_emb_dim)
            self.null_token = num_classes   # index used for unconditional pass
        else:
            self.class_emb = None
            cond_emb_dim   = None           # disables cond projection in ResidualBlocks

        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        # --- Encoder ---
        self.init_conv = nn.Conv2d(img_channels, dims[0], 3, padding=1)
        self.down1 = ResidualBlock(dims[0], dims[1], time_emb_dim, cond_emb_dim)
        self.down2 = ResidualBlock(dims[1], dims[2], time_emb_dim, cond_emb_dim)
        self.down3 = ResidualBlock(dims[2], dims[3], time_emb_dim, cond_emb_dim)
        self.pool  = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.mid1 = ResidualBlock(dims[3], dims[3], time_emb_dim, cond_emb_dim)
        self.attn  = AttentionBlock(dims[3])
        self.mid2  = ResidualBlock(dims[3], dims[3], time_emb_dim, cond_emb_dim)

        # --- Decoder ---
        self.up3 = ResidualBlock(dims[3] + dims[3], dims[2], time_emb_dim, cond_emb_dim)
        self.up2 = ResidualBlock(dims[2] + dims[2], dims[1], time_emb_dim, cond_emb_dim)
        self.up1 = ResidualBlock(dims[1] + dims[1], dims[0], time_emb_dim, cond_emb_dim)
        self.up0 = ResidualBlock(dims[0] + dims[0], dims[0], time_emb_dim, cond_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, dims[0]),
            nn.SiLU(),
            nn.Conv2d(dims[0], img_channels, 1)
        )

    @staticmethod
    def _match_size(upsampled, skip):
        """Ensure upsampled H,W exactly matches skip's H,W before cat."""
        _, _, H, W = skip.shape
        return F.interpolate(upsampled, size=(H, W), mode="nearest")

    def forward(self, x, t, c=None):
        """
        Args:
            x: (B, C, H, W) noisy image.
            t: (B,) timestep indices.
            c: (B,) class label indices, or None for unconditional.
        """
        t_emb = self.time_mlp(t)

        # Build conditioning embedding (None if model is unconditional)
        c_emb = self.class_emb(c) if (self.class_emb is not None and c is not None) else None

        # Encoder
        x0 = self.init_conv(x)
        x1 = self.down1(self.pool(x0), t_emb, c_emb)
        x2 = self.down2(self.pool(x1), t_emb, c_emb)
        x3 = self.down3(self.pool(x2), t_emb, c_emb)

        # Bottleneck
        h = self.mid1(x3, t_emb, c_emb)
        h = self.attn(h)
        h = self.mid2(h, t_emb, c_emb)

        # Decoder
        h = self.up3(torch.cat([self._match_size(F.interpolate(h, scale_factor=2), x3), x3], dim=1), t_emb, c_emb)
        h = self.up2(torch.cat([self._match_size(F.interpolate(h, scale_factor=2), x2), x2], dim=1), t_emb, c_emb)
        h = self.up1(torch.cat([self._match_size(F.interpolate(h, scale_factor=2), x1), x1], dim=1), t_emb, c_emb)
        h = self.up0(torch.cat([self._match_size(F.interpolate(h, scale_factor=2), x0), x0], dim=1), t_emb, c_emb)

        return self.out_conv(h)
