"""
Encodeur texte pré-entraîné (CLIP), gelé, pour le conditionnement du U-Net.

Pourquoi un modèle pré-entraîné et pas un encodeur "from scratch" ?
Comprendre du texte libre (grammaire, synonymes, combinaisons d'attributs)
demande d'avoir vu énormément de langage — bien plus que ce que nos ~200k
captions CelebA-Dialog peuvent apprendre à un petit encodeur entraîné de
zéro. On réutilise donc CLIP (Radford et al., 2021), comme le fait Stable
Diffusion, mais on ne met PAS à jour ses poids (`requires_grad=False`) :
seul notre U-Net apprendra, à partir des embeddings que CLIP produit.

CLIP transforme une phrase en une séquence de vecteurs (un par token), ce
qui permettra plus tard au U-Net de faire du cross-attention sur le texte
(plutôt qu'un unique vecteur global, ce qui serait plus pauvre).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

# Checkpoint CLIP standard (le même que celui utilisé par Stable Diffusion 1.x).
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class FrozenCLIPTextEncoder(nn.Module):
    """Encode une liste de phrases en séquence d'embeddings, via CLIP gelé.

    Args:
        model_name: checkpoint HuggingFace du text encoder CLIP.
        max_length: longueur max de tokenisation (les captions plus longues sont tronquées).
        device: device torch sur lequel placer le modèle.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        max_length: int = 77,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        from src.backend.shared.device import get_device

        self.device = torch.device(device) if device is not None else get_device()
        self.max_length = max_length

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device)

        # Gèle CLIP : on ne veut jamais backpropager dans ses poids.
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Dimension des embeddings produits par CLIP (768 pour ViT-B/32),
        # utile pour dimensionner les couches du U-Net qui liront ce texte.
        self.embed_dim = self.text_model.config.hidden_size

    @torch.no_grad()
    def forward(self, captions: list[str]) -> torch.Tensor:
        """Encode une liste de phrases.

        Args:
            captions: liste de B phrases.

        Returns:
            Tenseur (B, seq_len, embed_dim) : un embedding par token de chaque
            phrase (padding inclus), prêt pour du cross-attention dans le U-Net.
        """
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_model(**tokens)
        return outputs.last_hidden_state  # (B, seq_len, embed_dim)


if __name__ == "__main__":
    # Auto-test : encode quelques vraies captions CelebA et vérifie les shapes.
    from src.backend.shared.dataset import CelebACaptionDataset

    encoder = FrozenCLIPTextEncoder()
    print(f"Device: {encoder.device}")
    print(f"Dimension des embeddings CLIP: {encoder.embed_dim}")

    # Vérifie que les poids sont bien gelés.
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Nombre de paramètres entraînables dans CLIP: {n_trainable} (doit être 0)")

    ds = CelebACaptionDataset(split="train", image_size=64)
    captions = [ds[i][1] for i in range(4)]
    print("\nCaptions test:")
    for c in captions:
        print(f"  - {c!r}")

    embeddings = encoder(captions)
    print(f"\nShape des embeddings: {tuple(embeddings.shape)}  (B, seq_len, embed_dim)")
    print(f"Device des embeddings: {embeddings.device}")
