"""
Figure pédagogique : champ de vitesse (vector field) appris par un modèle de
Flow Matching, illustré sur un exemple jouet 2D avec DEUX points de données
cibles (pour montrer que le champ peut être multimodal, comme dans le cas
réel où x_0 est tirée d'une vraie distribution de données, pas un point unique).

On affiche ici le champ de GÉNÉRATION, c'est-à-dire -v_theta(x, t) : le sens
réellement suivi par le solveur ODE pendant le sampling (x part du bruit et
"redescend" vers les données, x_{t-dt} = x_t - dt * v_theta(x_t, t)). Les
flèches convergent donc visuellement vers les modes de données, ce qui est
plus intuitif à lire qu'une représentation brute de v_theta = noise - x_0
(qui pointerait, elle, dans le sens opposé).
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(3)

# Deux "modes" de données cibles (illustrant une distribution multimodale).
targets = np.array([[2.2, 1.8], [-1.8, 2.0]])

t_fixed = 0.5  # instant auquel on visualise le champ

grid = np.linspace(-3.5, 3.5, 18)
X, Y = np.meshgrid(grid, grid)

U = np.zeros_like(X)
V = np.zeros_like(Y)

# Pour chaque point de la grille (interprété comme x_t à l'instant t_fixed),
# on calcule la vélocité "moyenne pondérée" vers les cibles les plus proches
# (approximation simplifiée du champ de vitesse marginal appris par le modèle).
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        weights = []
        directions = []
        for target in targets:
            # À l'instant t, un point interpolé depuis `target` serait à
            # (1-t)*target + t*noise. On pondère par la proximité inverse.
            dist = np.linalg.norm(point - target)
            weight = np.exp(-dist**2 / 2.0)
            weights.append(weight)
            directions.append(point - target)  # direction "s'éloigner de la cible" = vers le bruit
        weights = np.array(weights)
        weights /= weights.sum() + 1e-8
        velocity = sum(w * d for w, d in zip(weights, directions))
        # On affiche -velocity (sens du sampling, vers les données) plutôt
        # que v_theta lui-même (sens du bruitage) : plus intuitif à lire.
        generation_direction = -velocity
        norm = np.linalg.norm(generation_direction) + 1e-8
        U[i, j] = generation_direction[0] / norm
        V[i, j] = generation_direction[1] / norm

fig, ax = plt.subplots(figsize=(6.4, 5.6))

ax.quiver(X, Y, U, V, color="#4f46e5", alpha=0.65, width=0.0035)
ax.scatter(targets[:, 0], targets[:, 1], color="#059669", s=140, zorder=5, label=r"modes de $x_0$ (données)")

for k, target in enumerate(targets):
    ax.annotate(f"mode {k+1}", target, textcoords="offset points", xytext=(10, 8), fontsize=9)

ax.set_title(r"Champ de génération $-v_\theta(x, t)$ (schématique, $t=0.5$)")
ax.set_xlabel("dimension 1 (schématique)")
ax.set_ylabel("dimension 2 (schématique)")
ax.legend(loc="upper right", fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("report/latex/figures/vector_field.pdf")
print("saved vector_field.pdf")
