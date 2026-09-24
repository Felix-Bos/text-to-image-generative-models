"""
Sauvegarde d'une grille d'images générées pendant l'entraînement, pour
suivre visuellement la progression (utilisé par diffusion/train.py et
flow_matching/train.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_preview_grid(
    images: torch.Tensor,
    captions: list[str],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Sauvegarde une grille d'images (normalisées [-1, 1]) avec leurs captions.

    Args:
        images: shape (B, C, H, W), valeurs approx. dans [-1, 1].
        captions: une légende par image (affichée comme titre de sous-figure).
        out_path: chemin du fichier .png à écrire (le dossier parent est créé si besoin).
        title: titre global optionnel de la figure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = images.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.3))
    if n == 1:
        axes = [axes]

    for ax, img, caption in zip(axes, images, captions):
        img_to_show = (img.clamp(-1, 1) + 1) / 2
        ax.imshow(img_to_show.cpu().permute(1, 2, 0).numpy())
        ax.set_title(caption, fontsize=7, wrap=True)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
