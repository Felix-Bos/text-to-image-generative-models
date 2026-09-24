"""
Time embedding pour le U-Net de diffusion.

Le U-Net doit savoir à quel timestep t (niveau de bruit) il travaille, pour
chaque image du batch. Un entier brut serait un signal pauvre à apprendre ;
on le transforme donc en vecteur via un encodage sinusoïdal (le même principe
que le positional encoding des Transformers), puis un petit MLP lui donne
plus de capacité d'expression.

    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))

Deux timesteps proches produisent des embeddings proches (continuité), et le
réseau peut apprendre à distinguer n'importe quel timestep parmi les T.

Le vecteur produit ici (dimension `time_embed_dim`) sera ensuite injecté dans
chaque ResBlock du U-Net (voir unet_blocks.py), pas seulement à l'entrée.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode un batch de timesteps en vecteurs sinusoïdaux.

    Args:
        t: timesteps, shape (B,), valeurs entières (ou float).
        dim: dimension de l'embedding produit (doit être pair).

    Returns:
        Tenseur (B, dim).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim doit être pair, reçu {dim}")

    half_dim = dim // 2
    # Fréquences décroissantes exponentiellement, comme dans "Attention Is All You Need".
    freq_exponents = torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
    inv_freq = torch.exp(-math.log(10000.0) * freq_exponents)  # (half_dim,)

    # (B, 1) * (half_dim,) -> (B, half_dim) via broadcasting
    angles = t.float().unsqueeze(1) * inv_freq.unsqueeze(0)

    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
    return embedding


class TimeEmbedding(nn.Module):
    """Encodage sinusoïdal de t, suivi d'un MLP (Linear -> SiLU -> Linear).

    Args:
        base_dim: dimension de l'encodage sinusoïdal brut.
        time_embed_dim: dimension finale du vecteur temps (après le MLP),
            celle qui sera effectivement injectée dans les ResBlocks.
    """

    def __init__(self, base_dim: int = 128, time_embed_dim: int = 512) -> None:
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.base_dim)  # (B, base_dim)
        return self.mlp(emb)  # (B, time_embed_dim)


if __name__ == "__main__":
    # Auto-test : vérifie les shapes et visualise l'embedding sinusoïdal brut.
    import matplotlib.pyplot as plt

    from src.backend.shared.device import get_device

    device = get_device()

    # --- Sanity check sur sinusoidal_embedding seule ---
    t = torch.arange(0, 1000, device=device)
    raw_emb = sinusoidal_embedding(t, dim=128)
    print(f"raw_emb shape: {tuple(raw_emb.shape)}  (T, base_dim)")
    print(f"valeurs min/max: {raw_emb.min().item():.3f} / {raw_emb.max().item():.3f} (doivent être dans [-1, 1])")

    # --- Sanity check sur le module complet (avec MLP) ---
    time_embed = TimeEmbedding(base_dim=128, time_embed_dim=512).to(device)
    t_batch = torch.tensor([0, 100, 500, 999], device=device)
    out = time_embed(t_batch)
    print(f"\nTimeEmbedding output shape: {tuple(out.shape)}  (B, time_embed_dim)")

    # Deux timesteps proches doivent produire des embeddings proches (continuité).
    t_close = torch.tensor([500, 501], device=device)
    emb_close = time_embed(t_close)
    dist_close = (emb_close[0] - emb_close[1]).norm().item()

    t_far = torch.tensor([0, 999], device=device)
    emb_far = time_embed(t_far)
    dist_far = (emb_far[0] - emb_far[1]).norm().item()

    print(f"\nDistance entre embeddings de t=500 et t=501 (proches): {dist_close:.3f}")
    print(f"Distance entre embeddings de t=0 et t=999 (éloignés):    {dist_far:.3f}")
    print("(la distance 'proches' doit être nettement plus petite que 'éloignés')")

    # --- Visualisation de l'encodage sinusoïdal brut (avant MLP) ---
    plt.figure(figsize=(8, 5))
    plt.imshow(raw_emb.cpu().numpy().T, aspect="auto", cmap="RdBu")
    plt.xlabel("timestep t")
    plt.ylabel("dimension de l'embedding")
    plt.title("Encodage sinusoïdal de t (avant MLP)")
    plt.colorbar()
    plt.tight_layout()
    out_path = "report/time_embedding_example.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure sauvegardée dans {out_path}")
