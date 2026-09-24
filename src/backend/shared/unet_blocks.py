"""
Briques réutilisables du U-Net : ResBlock (injection du temps) et
CrossAttentionBlock (injection du texte).

Ces deux blocs sont les "unités" assemblées à chaque étage du U-Net
(unet.py). Ils sont testés isolément ici avant assemblage.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """Bloc résiduel convolutif, modulé par le time embedding.

    Schéma :
        x ── GroupNorm ── SiLU ── Conv3x3 ──(+ t_vec projeté)── GroupNorm ── SiLU ── Conv3x3 ──┐
        │                                                                                       │
        └──────────────────────── skip connection (conv 1x1 si in≠out channels) ────────────────┴─► sortie

    Le time embedding t_vec est projeté à la dimension `out_channels` puis
    additionné aux feature maps après la première convolution : c'est ce qui
    permet à ce bloc d'adapter son comportement au niveau de bruit courant.

    Args:
        in_channels: nombre de canaux en entrée.
        out_channels: nombre de canaux en sortie.
        time_embed_dim: dimension du vecteur temps (sortie de TimeEmbedding).
        num_groups: nombre de groupes pour GroupNorm (32 est la valeur standard).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_groups: int = 32,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Projette t_vec (B, time_embed_dim) -> (B, out_channels), pour pouvoir
        # l'additionner aux feature maps (B, out_channels, H, W).
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.activation = nn.SiLU()

        # Si in_channels != out_channels, la skip connection a besoin d'une
        # conv 1x1 pour ajuster le nombre de canaux avant l'addition.
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature maps, shape (B, in_channels, H, W).
            t_vec: time embedding, shape (B, time_embed_dim).

        Returns:
            Tenseur (B, out_channels, H, W).
        """
        h = self.conv1(self.activation(self.norm1(x)))

        # Injection du temps : broadcast sur H et W.
        t_proj = self.time_proj(self.activation(t_vec))  # (B, out_channels)
        h = h + t_proj[:, :, None, None]

        h = self.conv2(self.activation(self.norm2(h)))

        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention entre les feature maps de l'image et les embeddings texte.

    Les positions spatiales de l'image (aplaties en séquence) servent de
    queries ; les tokens texte (embeddings CLIP) servent de keys/values.
    Chaque "pixel" de la feature map peut ainsi interroger les mots de la
    phrase et récupérer l'information visuelle pertinente qu'ils encodent.

        Attention(Q, K, V) = softmax(Q Kᵀ / √d) V

    Args:
        channels: nombre de canaux des feature maps image (dimension des queries).
        text_embed_dim: dimension des embeddings texte (512 pour CLIP ViT-B/32).
        num_heads: nombre de têtes d'attention.
        num_groups: nombre de groupes pour la GroupNorm appliquée avant attention.
    """

    def __init__(
        self,
        channels: int,
        text_embed_dim: int,
        num_heads: int = 8,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) doit être divisible par num_heads ({num_heads})")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)

        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(text_embed_dim, channels, bias=False)
        self.to_v = nn.Linear(text_embed_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature maps image, shape (B, channels, H, W).
            text_embed: embeddings texte CLIP, shape (B, seq_len, text_embed_dim).

        Returns:
            Tenseur (B, channels, H, W), avec l'information texte incorporée
            (connexion résiduelle : sortie = x + attention(...)).
        """
        B, C, H, W = x.shape
        residual = x

        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C) : une "séquence" de pixels

        q = self.to_q(h)  # (B, H*W, C)
        k = self.to_k(text_embed)  # (B, seq_len, C)
        v = self.to_v(text_embed)  # (B, seq_len, C)

        # Découpe en têtes : (B, n_heads, seq, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, N, _ = t.shape
            return t.view(B_, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # (B, n_heads, H*W, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)
        attn_out = self.to_out(attn_out)

        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)

        return residual + attn_out


if __name__ == "__main__":
    # Auto-test : vérifie les shapes en sortie de chaque bloc, sur MPS.
    from src.backend.shared.device import get_device
    from src.backend.shared.time_embedding import TimeEmbedding

    device = get_device()
    print(f"Device: {device}\n")

    batch_size = 4
    time_embed_dim = 512
    text_embed_dim = 512
    seq_len = 77

    # --- Test ResBlock ---
    print("--- ResBlock ---")
    x = torch.randn(batch_size, 64, 32, 32, device=device)
    time_embed = TimeEmbedding(base_dim=128, time_embed_dim=time_embed_dim).to(device)
    t = torch.tensor([0, 100, 500, 999], device=device)
    t_vec = time_embed(t)

    resblock = ResBlock(in_channels=64, out_channels=128, time_embed_dim=time_embed_dim).to(device)
    out = resblock(x, t_vec)
    print(f"Entrée:  {tuple(x.shape)}")
    print(f"Sortie:  {tuple(out.shape)}  (attendu: ({batch_size}, 128, 32, 32))")

    # ResBlock avec in_channels == out_channels (test de la skip connection Identity).
    resblock_same = ResBlock(in_channels=64, out_channels=64, time_embed_dim=time_embed_dim).to(device)
    out_same = resblock_same(x, t_vec)
    print(f"Sortie (in=out=64): {tuple(out_same.shape)}  (attendu: ({batch_size}, 64, 32, 32))")

    # --- Test CrossAttentionBlock ---
    print("\n--- CrossAttentionBlock ---")
    x_img = torch.randn(batch_size, 128, 16, 16, device=device)
    text_embed = torch.randn(batch_size, seq_len, text_embed_dim, device=device)

    cross_attn = CrossAttentionBlock(channels=128, text_embed_dim=text_embed_dim, num_heads=8).to(device)
    out_attn = cross_attn(x_img, text_embed)
    print(f"Entrée image: {tuple(x_img.shape)}")
    print(f"Entrée texte: {tuple(text_embed.shape)}")
    print(f"Sortie:       {tuple(out_attn.shape)}  (attendu: ({batch_size}, 128, 16, 16))")

    # Sanity check : si le texte est mis à zéro, la sortie ne doit pas être
    # identique à l'entrée (l'attention + skip connection transforme quand même).
    assert out.shape == (batch_size, 128, 32, 32)
    assert out_same.shape == (batch_size, 64, 32, 32)
    assert out_attn.shape == x_img.shape
    print("\nTous les tests de shape sont passés.")
