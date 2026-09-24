"""
Génération d'images par Flow Matching : résout l'équation différentielle
dx/dt = v_pred(x, t) à rebours, de t=1 (bruit pur) à t=0 (image).

Contrairement au DDPM (processus stochastique, un bruit gaussien est ajouté
à chaque étape), la génération par flow matching est déterministe : à un
bruit de départ fixé, la trajectoire est entièrement déterminée par le champ
de vitesse appris.

On utilise ici le solveur le plus simple possible, Euler explicite :

    x_{t - dt} = x_t - dt * v_pred(x_t, t)

avec un nombre d'étapes `num_steps` généralement bien plus petit que les T=1000
étapes utilisées par le DDPM (le chemin étant une droite, moins de pas
suffisent en pratique pour bien l'approximer).
"""

from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm


@torch.no_grad()
def generate(
    unet: nn.Module,
    text_embed: torch.Tensor,
    image_size: int = 64,
    channels: int = 3,
    num_steps: int = 50,
    device: str | torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Génère un batch d'images conditionnées par `text_embed`, via un
    solveur ODE Euler explicite, en partant de bruit pur (t=1).

    Args:
        unet: le U-Net entraîné à prédire le champ de vitesse (même
            architecture que pour le DDPM, voir shared/unet.py).
        text_embed: embeddings texte CLIP, shape (B, seq_len, embed_dim).
        image_size: taille (carrée) des images générées.
        channels: nombre de canaux (3 pour RGB).
        num_steps: nombre de pas du solveur ODE (moins que le DDPM en général).
        device: device torch.
        show_progress: affiche une barre de progression.

    Returns:
        Images générées, shape (B, channels, image_size, image_size), valeurs
        dans [-1, 1] environ (pas de garantie stricte, contrairement au DDPM).
    """
    device = torch.device(device) if device is not None else next(unet.parameters()).device
    batch_size = text_embed.shape[0]

    unet.eval()

    # Point de départ : t=1 (bruit gaussien pur).
    x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)

    dt = 1.0 / num_steps
    # On part de t=1 et on descend vers t=0 par pas de dt.
    time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]

    iterator = time_steps
    if show_progress:
        iterator = tqdm(time_steps, desc="Sampling Flow Matching")

    for t_val in iterator:
        t = torch.full((batch_size,), t_val.item(), device=device)
        v_pred = unet(x_t, t, text_embed)
        x_t = x_t - dt * v_pred  # pas d'Euler explicite, en remontant le temps

    return x_t


if __name__ == "__main__":
    # Auto-test : génère un batch d'images à partir d'un U-Net NON entraîné.
    # Vérifie seulement que la boucle ODE tourne et produit des tenseurs finis.
    import matplotlib.pyplot as plt

    from src.backend.shared.device import get_device
    from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
    from src.backend.shared.unet import UNet

    device = get_device()
    print(f"Device: {device}")

    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)

    text_encoder = FrozenCLIPTextEncoder(device=device)
    captions = ["a smiling young woman with bangs", "a man with a beard and glasses"]
    text_embed = text_encoder(captions)

    images = generate(unet, text_embed, image_size=64, num_steps=50, device=device)
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
    out_path = "report/flow_matching_sample_untrained.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure sauvegardée dans {out_path}")
    print("(U-Net non entraîné : du bruit est attendu, ceci valide seulement la boucle ODE)")
