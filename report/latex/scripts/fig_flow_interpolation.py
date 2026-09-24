"""Régénère la figure d'interpolation Flow Matching en PDF pour le rapport LaTeX."""

import matplotlib.pyplot as plt
import torch

from src.backend.flow_matching.flow_matching import ConditionalFlowMatching
from src.backend.shared.dataset import CelebACaptionDataset

flow = ConditionalFlowMatching(device="cpu")

ds = CelebACaptionDataset(split="train", image_size=64)
image, _caption = ds[0]
x_0 = image.unsqueeze(0)

torch.manual_seed(0)
fixed_noise = torch.randn_like(x_0)

steps_to_show = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
fig, axes = plt.subplots(1, len(steps_to_show), figsize=(2.1 * len(steps_to_show), 2.3))

for ax, t_val in zip(axes, steps_to_show):
    t = torch.tensor([t_val])
    x_t, _, _ = flow.sample_xt(x_0, t, noise=fixed_noise)
    img_to_show = (x_t[0].clamp(-1, 1) + 1) / 2
    ax.imshow(img_to_show.permute(1, 2, 0).numpy())
    ax.set_title(f"$t={t_val:.2f}$", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("report/latex/figures/flow_interpolation.pdf")
print("saved flow_interpolation.pdf")
