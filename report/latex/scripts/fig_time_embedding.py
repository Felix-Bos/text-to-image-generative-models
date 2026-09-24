"""Régénère la figure d'encodage sinusoïdal du temps en PDF pour le rapport LaTeX."""

import matplotlib.pyplot as plt
import torch

from src.backend.shared.time_embedding import sinusoidal_embedding

t = torch.arange(0, 1000)
raw_emb = sinusoidal_embedding(t, dim=128)

plt.figure(figsize=(7, 4.2))
plt.imshow(raw_emb.numpy().T, aspect="auto", cmap="RdBu")
plt.xlabel(r"timestep $t$")
plt.ylabel("dimension de l'embedding")
plt.title(r"Encodage sinusoïdal de $t$ (avant MLP)")
plt.colorbar()
plt.tight_layout()
plt.savefig("report/latex/figures/time_embedding.pdf")
print("saved time_embedding.pdf")
