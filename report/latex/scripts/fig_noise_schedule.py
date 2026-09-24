"""Génère la figure du schedule de bruit DDPM (beta_t, alpha_t, alpha_bar_t)."""

import matplotlib.pyplot as plt
import numpy as np

T = 1000
beta_start, beta_end = 1e-4, 2e-2

betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

axes[0].plot(betas, color="#4f46e5")
axes[0].set_title(r"$\beta_t$ (bruit ajouté à l'étape $t$)")
axes[0].set_xlabel(r"$t$")
axes[0].grid(alpha=0.3)

axes[1].plot(alphas, color="#059669")
axes[1].set_title(r"$\alpha_t = 1 - \beta_t$")
axes[1].set_xlabel(r"$t$")
axes[1].grid(alpha=0.3)

axes[2].plot(alphas_cumprod, color="#dc2626")
axes[2].set_title(r"$\bar\alpha_t = \prod_{s=1}^t \alpha_s$")
axes[2].set_xlabel(r"$t$")
axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=1)
axes[2].axhline(0.0, color="gray", linestyle=":", linewidth=1)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("report/latex/figures/noise_schedule.pdf")
print("saved noise_schedule.pdf")
