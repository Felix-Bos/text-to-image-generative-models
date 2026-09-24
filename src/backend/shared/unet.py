"""
U-Net complet pour le DDPM text-to-image.

Assemble les briques des modules précédents :
    - time_embedding.py : transforme t en vecteur (TimeEmbedding)
    - unet_blocks.py     : ResBlock (injection du temps) et CrossAttentionBlock (injection du texte)

Architecture (forme en "U", comme un U-Net de segmentation) :

    x_t (B,3,64,64)
          │
     Conv d'entrée
          │
    ┌─────▼─────┐  64x64   : ResBlock x2                (pas de cross-attn, trop coûteux à cette résolution)
    │  down[0]  │
    └─────┬─────┘
          │ downsample -> 32x32
    ┌─────▼─────┐  32x32   : ResBlock x2 + CrossAttn
    │  down[1]  │
    └─────┬─────┘
          │ downsample -> 16x16
    ┌─────▼─────┐  16x16   : ResBlock x2 + CrossAttn
    │  down[2]  │
    └─────┬─────┘
          │ downsample -> 8x8
    ┌─────▼─────┐  8x8     : ResBlock + CrossAttn + ResBlock  (bottleneck)
    │ bottleneck│
    └─────┬─────┘
          │ upsample -> 16x16, + skip connection depuis down[2]
    ┌─────▼─────┐
    │   up[2]   │  ResBlock x2 + CrossAttn
    └─────┬─────┘
          │ upsample -> 32x32, + skip connection depuis down[1]
    ┌─────▼─────┐
    │   up[1]   │  ResBlock x2 + CrossAttn
    └─────┬─────┘
          │ upsample -> 64x64, + skip connection depuis down[0]
    ┌─────▼─────┐
    │   up[0]   │  ResBlock x2  (pas de cross-attn)
    └─────┬─────┘
          │
     Conv de sortie
          │
    ε_prédit (B,3,64,64)

t est injecté dans CHAQUE ResBlock (à tous les étages). Le texte est injecté
via cross-attention aux résolutions 32x32, 16x16 et 8x8 (pas à 64x64, où le
coût de l'attention (O(H²W²)) serait trop élevé pour le bénéfice apporté).
"""

from __future__ import annotations

import torch
from torch import nn

from src.backend.shared.time_embedding import TimeEmbedding
from src.backend.shared.unet_blocks import CrossAttentionBlock, ResBlock


class Downsample(nn.Module):
    """Divise la résolution spatiale par 2 (conv stride 2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Multiplie la résolution spatiale par 2 (interpolation + conv)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownStage(nn.Module):
    """Un étage de la descente : 2 ResBlocks (+ CrossAttention optionnelle)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        text_embed_dim: int,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        self.res1 = ResBlock(in_channels, out_channels, time_embed_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_embed_dim)
        if use_attention:
            self.attn1 = CrossAttentionBlock(out_channels, text_embed_dim)
            self.attn2 = CrossAttentionBlock(out_channels, text_embed_dim)

    def forward(
        self, x: torch.Tensor, t_vec: torch.Tensor, text_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t_vec)
        if self.use_attention:
            x = self.attn1(x, text_embed)
        x = self.res2(x, t_vec)
        if self.use_attention:
            x = self.attn2(x, text_embed)
        # On retourne x deux fois : une copie sera stockée comme skip connection.
        return x, x


class UpStage(nn.Module):
    """Un étage de la remontée : reçoit la skip connection, 2 ResBlocks (+ CrossAttention optionnelle)."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embed_dim: int,
        text_embed_dim: int,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        # La skip connection est concaténée sur les canaux avant le premier ResBlock.
        self.res1 = ResBlock(in_channels + skip_channels, out_channels, time_embed_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_embed_dim)
        if use_attention:
            self.attn1 = CrossAttentionBlock(out_channels, text_embed_dim)
            self.attn2 = CrossAttentionBlock(out_channels, text_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_vec: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_vec)
        if self.use_attention:
            x = self.attn1(x, text_embed)
        x = self.res2(x, t_vec)
        if self.use_attention:
            x = self.attn2(x, text_embed)
        return x


class UNet(nn.Module):
    """U-Net de débruitage, conditionné par le temps t et un texte (embeddings CLIP).

    Args:
        in_channels: nombre de canaux de l'image en entrée (3 pour RGB).
        base_channels: nombre de canaux au premier étage (les étages suivants
            multiplient ce nombre selon `channel_multipliers`).
        channel_multipliers: multiplicateurs de canaux à chaque étage de résolution.
        time_embed_dim: dimension du vecteur temps (sortie de TimeEmbedding).
        text_embed_dim: dimension des embeddings texte (512 pour CLIP ViT-B/32).
        attention_resolutions: indices d'étages (0 = résolution la plus haute)
            où la cross-attention texte est activée.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        time_embed_dim: int = 512,
        text_embed_dim: int = 512,
        attention_resolutions: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()

        self.time_embedding = TimeEmbedding(base_dim=128, time_embed_dim=time_embed_dim)

        channels = [base_channels * m for m in channel_multipliers]
        # ex: base_channels=64, multipliers=(1,2,4,4) -> channels = [64, 128, 256, 256]

        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # --- Down path ---
        self.down_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            use_attn = i in attention_resolutions
            self.down_stages.append(
                DownStage(in_ch, out_ch, time_embed_dim, text_embed_dim, use_attn)
            )
            in_ch = out_ch
            # Pas de downsample après le dernier étage (on est déjà au bottleneck).
            if i < len(channels) - 1:
                self.downsamplers.append(Downsample(out_ch))

        # --- Bottleneck ---
        bottleneck_ch = channels[-1]
        self.bottleneck_res1 = ResBlock(bottleneck_ch, bottleneck_ch, time_embed_dim)
        self.bottleneck_attn = CrossAttentionBlock(bottleneck_ch, text_embed_dim)
        self.bottleneck_res2 = ResBlock(bottleneck_ch, bottleneck_ch, time_embed_dim)

        # --- Up path (symétrique du down path, ordre inversé) ---
        self.up_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        in_ch = bottleneck_ch
        reversed_channels = list(reversed(channels))
        for i, out_ch in enumerate(reversed_channels):
            stage_idx = len(channels) - 1 - i  # index correspondant dans le down path
            use_attn = stage_idx in attention_resolutions
            skip_ch = channels[stage_idx]
            self.up_stages.append(
                UpStage(in_ch, skip_ch, out_ch, time_embed_dim, text_embed_dim, use_attn)
            )
            in_ch = out_ch
            if i < len(channels) - 1:
                self.upsamplers.append(Upsample(out_ch))

        self.norm_out = nn.GroupNorm(32, channels[0])
        self.conv_out = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: image bruitée, shape (B, in_channels, H, W).
            t: timesteps, shape (B,).
            text_embed: embeddings texte CLIP, shape (B, seq_len, text_embed_dim).

        Returns:
            Bruit prédit, shape (B, in_channels, H, W) (même forme que x).
        """
        t_vec = self.time_embedding(t)

        h = self.conv_in(x)

        # --- Down path : on garde chaque sortie d'étage pour les skip connections ---
        skips = []
        for i, stage in enumerate(self.down_stages):
            h, skip = stage(h, t_vec, text_embed)
            skips.append(skip)
            if i < len(self.downsamplers):
                h = self.downsamplers[i](h)

        # --- Bottleneck ---
        h = self.bottleneck_res1(h, t_vec)
        h = self.bottleneck_attn(h, text_embed)
        h = self.bottleneck_res2(h, t_vec)

        # --- Up path : consomme les skips dans l'ordre inverse ---
        for i, stage in enumerate(self.up_stages):
            skip = skips.pop()
            h = stage(h, skip, t_vec, text_embed)
            if i < len(self.upsamplers):
                h = self.upsamplers[i](h)

        h = self.conv_out(self.activation(self.norm_out(h)))
        return h


if __name__ == "__main__":
    # Auto-test : forward pass complet sur des données factices, puis sur une
    # vraie image + caption du dataset (avec CLIP), pour vérifier l'intégration
    # de bout en bout.
    from src.backend.shared.device import get_device

    device = get_device()
    print(f"Device: {device}\n")

    # --- Test 1 : shapes avec des tenseurs aléatoires ---
    print("--- Test 1 : forward pass avec données factices ---")
    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"Nombre de paramètres du U-Net: {n_params:,}")

    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    text_embed = torch.randn(batch_size, 77, 512, device=device)

    out = unet(x, t, text_embed)
    print(f"Entrée x:      {tuple(x.shape)}")
    print(f"Sortie ε_pred: {tuple(out.shape)}  (doit être identique à x)")
    assert out.shape == x.shape

    print(
        "\nCe U-Net est partagé entre diffusion/ et flow_matching/ : il ne sait rien de "
        "l'objectif d'entraînement (bruit vs champ de vitesse), il apprend juste à mapper "
        "(x_t, t, texte) -> sortie de même forme que x_t. Les tests d'intégration complets "
        "(avec le vrai dataset, CLIP, et le bruitage/interpolation spécifique à chaque "
        "approche) sont dans src/backend/diffusion/train.py et src/backend/flow_matching/train.py."
    )
    print("\nTest de shape passé.")
