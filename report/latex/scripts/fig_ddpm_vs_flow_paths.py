"""
Figure pédagogique : compare, sur un exemple jouet en 2D, les trajectoires
suivies par un DDPM (marche stochastique, schedule non-linéaire) et par un
Flow Matching (segment de droite déterministe) entre un même point de départ
(bruit) et un même point cible (donnée).

Ce n'est PAS une simulation du vrai modèle entraîné : c'est une illustration
géométrique du PRINCIPE de chaque approche, pour l'intuition.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

x0 = np.array([2.5, 2.0])  # point "donnée" (image cible)
x1 = np.array([-2.0, -2.5])  # point "bruit" de départ

T = 60

# --- Trajectoire Flow Matching : interpolation linéaire déterministe ---
ts = np.linspace(0, 1, T)
flow_path = np.array([(1 - t) * x0 + t * x1 for t in ts])

# --- Trajectoire DDPM : marche avec bruit à chaque étape, schedule non-linéaire ---
# On simule un chemin qui suit approximativement le même schedule cumulatif
# que le vrai DDPM (alpha_bar_t), mais en ajoutant du bruit latéral à chaque
# pas pour illustrer le caractère stochastique du processus.
beta_start, beta_end = 1e-4, 2e-2
betas = np.linspace(beta_start, beta_end, T)
alphas_cumprod = np.cumprod(1 - betas)
# alphas_cumprod décroît de ~1 à ~0 : on l'utilise comme "combien de x0 reste"
ddpm_path = np.zeros((T, 2))
for i, ac in enumerate(alphas_cumprod):
    mean = np.sqrt(ac) * x0 + np.sqrt(1 - ac) * x1
    lateral_noise = rng.normal(scale=0.18, size=2) * (1 - abs(2 * i / T - 1))  # bruit plus fort au milieu
    ddpm_path[i] = mean + lateral_noise

fig, ax = plt.subplots(figsize=(6.2, 5.6))

ax.plot(flow_path[:, 0], flow_path[:, 1], color="#4f46e5", linewidth=2.2, label="Flow Matching (droite déterministe)")
ax.plot(ddpm_path[:, 0], ddpm_path[:, 1], color="#dc2626", linewidth=1.6, alpha=0.85, label="DDPM (marche stochastique)")

ax.scatter(*x0, color="#059669", s=110, zorder=5, label=r"$x_0$ (donnée)")
ax.scatter(*x1, color="#111827", s=110, zorder=5, label=r"$x_1$ (bruit)")

ax.annotate(r"$t=0$", x0, textcoords="offset points", xytext=(8, 8), fontsize=10)
ax.annotate(r"$t=1$", x1, textcoords="offset points", xytext=(8, -14), fontsize=10)

ax.set_xlabel(r"dimension 1 (schématique)")
ax.set_ylabel(r"dimension 2 (schématique)")
ax.set_title("Chemins entre bruit et donnée : Flow Matching vs DDPM")
ax.legend(loc="upper left", fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("report/latex/figures/ddpm_vs_flow_paths.pdf")
print("saved ddpm_vs_flow_paths.pdf")
