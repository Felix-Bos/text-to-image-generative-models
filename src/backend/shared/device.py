"""Choix centralisé du device torch (mps > cuda > cpu).

Sur ce projet on tourne en local sur Mac (Apple Silicon), donc "mps" est le
device accéléré à privilégier quand il est disponible. `cuda` est gardé en
fallback pour le cas où le code tournerait un jour sur une machine avec GPU
NVIDIA (ex: si on reprend le projet sur Colab/un serveur).
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
