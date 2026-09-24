"""Régénère la figure de bruitage forward (DDPM) en PDF pour le rapport LaTeX."""

import matplotlib.pyplot as plt
import torch

from src.backend.diffusion.gaussian_diffusion import GaussianDiffusion
from src.backend.shared.dataset import CelebACaptionDataset

diffusion = GaussianDiffusion(timesteps=1000, device="cpu")

ds = CelebACaptionDataset(split="train", image_size=64)
image, _caption = ds[0]
x_0 = image.unsqueeze(0)

steps_to_show = [0, 50, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(1, len(steps_to_show), figsize=(2.1 * len(steps_to_show), 2.3))

torch.manual_seed(0)
for ax, t_val in zip(axes, steps_to_show):
    t = torch.tensor([t_val])
    x_t, _ = diffusion.q_sample(x_0, t)
    img_to_show = (x_t[0].clamp(-1, 1) + 1) / 2
    ax.imshow(img_to_show.permute(1, 2, 0).numpy())
    ax.set_title(f"$t={t_val}$", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("report/latex/figures/forward_diffusion.pdf")
print("saved forward_diffusion.pdf")
