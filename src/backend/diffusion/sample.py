"""
Génération d'images par DDPM : part de bruit pur et débruite pas à pas.

Boucle complète (voir gaussian_diffusion.py pour la théorie de p_sample) :

    x_T = bruit gaussien pur
    pour t = T-1, T-2, ..., 0 :
        eps_pred = U-Net(x_t, t, texte)
        x_{t-1}  = p_sample(eps_pred, x_t, t)
    x_0 = image finale générée

Le U-Net est appelé T fois (1000 par défaut) pour générer une seule image :
c'est le coût principal de la génération en DDPM (contrairement au flow
matching, qui peut généralement utiliser beaucoup moins d'étapes).
"""

from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm

from src.backend.diffusion.gaussian_diffusion import GaussianDiffusion


@torch.no_grad()
def generate(
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    text_embed: torch.Tensor,
    image_size: int = 64,
    channels: int = 3,
    device: str | torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Génère un batch d'images conditionnées par `text_embed`, depuis du bruit pur.

    Args:
        unet: le U-Net entraîné (voir shared/unet.py).
        diffusion: le schedule de bruit (voir gaussian_diffusion.py).
        text_embed: embeddings texte CLIP, shape (B, seq_len, embed_dim) —
            détermine à la fois le batch size et le conditionnement.
        image_size: taille (carrée) des images générées.
        channels: nombre de canaux (3 pour RGB).
        device: device torch (par défaut celui de `diffusion`).
        show_progress: affiche une barre de progression (T étapes).

    Returns:
        Images générées, shape (B, channels, image_size, image_size), valeurs
        dans [-1, 1] (même normalisation que le dataset d'entraînement).
    """
    device = torch.device(device) if device is not None else diffusion.device
    batch_size = text_embed.shape[0]

    unet.eval()

    x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)

    timesteps = reversed(range(diffusion.timesteps))
    if show_progress:
        timesteps = tqdm(timesteps, total=diffusion.timesteps, desc="Sampling DDPM")

    for t_val in timesteps:
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        eps_pred = unet(x_t, t, text_embed)
        x_t = diffusion.p_sample(eps_pred, x_t, t)

    return x_t


if __name__ == "__main__":
    # Auto-test : génère un batch d'images à partir d'un U-Net NON entraîné
    # (poids aléatoires). On ne s'attend évidemment pas à des visages, juste
    # à vérifier que la boucle de sampling tourne sans erreur et produit des
    # tenseurs bien formés/finis.
    import matplotlib.pyplot as plt

    from src.backend.shared.device import get_device
    from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
    from src.backend.shared.unet import UNet

    device = get_device()
    print(f"Device: {device}")

    # On réduit le nombre de timesteps pour que le test reste rapide (un vrai
    # entraînement utilisera 1000, mais valider la boucle n'a pas besoin d'autant).
    diffusion = GaussianDiffusion(timesteps=50, device=device)
    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)

    text_encoder = FrozenCLIPTextEncoder(device=device)
    captions = ["a smiling young woman with bangs", "a man with a beard and glasses"]
    text_embed = text_encoder(captions)

    images = generate(unet, diffusion, text_embed, image_size=64, device=device)
    print(f"\nImages générées: {tuple(images.shape)}")
    print(f"Valeurs min/max: {images.min().item():.3f} / {images.max().item():.3f}")
    assert torch.isfinite(images).all(), "Les images générées contiennent des NaN/Inf"

    fig, axes = plt.subplots(1, len(captions), figsize=(4 * len(captions), 4))
    for ax, img, caption in zip(axes, images, captions):
        img_to_show = (img.clamp(-1, 1) + 1) / 2
        ax.imshow(img_to_show.cpu().permute(1, 2, 0).numpy())
        ax.set_title(caption, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out_path = "report/ddpm_sample_untrained.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure sauvegardée dans {out_path}")
    print("(U-Net non entraîné : du bruit est attendu, ceci valide seulement la boucle de sampling)")
