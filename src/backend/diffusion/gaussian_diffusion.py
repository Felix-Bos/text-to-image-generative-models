"""
Forward process de diffusion (bruitage) pour un DDPM.

Théorie (Ho et al., 2020 - "Denoising Diffusion Probabilistic Models") :

On définit un schedule de bruit beta_1, ..., beta_T (croissant, petit -> plus grand).
On pose :
    alpha_t      = 1 - beta_t
    alpha_bar_t  = alpha_1 * alpha_2 * ... * alpha_t   (produit cumulé)

Grâce aux propriétés des lois gaussiennes, bruiter une image x_0 jusqu'à l'étape t
ne nécessite PAS de boucler t fois : une formule fermée suffit (le "reparameterization
trick" appliqué à la composition de gaussiennes) :

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

Intuition :
    - a t=0        : alpha_bar_t ~ 1        -> x_t ~ x_0 (image quasi intacte)
    - a t=T (grand) : alpha_bar_t ~ 0        -> x_t ~ eps (bruit gaussien pur)

C'est cette formule que ce module implémente. Le modèle (U-Net, à venir) apprendra
plus tard à inverser ce processus, c'est-à-dire à prédire `eps` à partir de `x_t`.
"""

from __future__ import annotations

import torch

from src.backend.shared.device import get_device


class GaussianDiffusion:
    """Gère le schedule de bruit et le bruitage forward (q(x_t | x_0)).

    Args:
        timesteps: nombre total d'étapes de diffusion T.
        beta_start: valeur de beta_1 (bruit ajouté à la première étape).
        beta_end: valeur de beta_T (bruit ajouté à la dernière étape).
        device: device torch sur lequel stocker les buffers du schedule.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device | None = None,
    ) -> None:
        self.timesteps = timesteps
        self.device = torch.device(device) if device is not None else get_device()

        # Schedule linéaire de beta_t, comme dans le papier original DDPM.
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)

        self.alphas = 1.0 - self.betas
        # Produit cumulé alpha_bar_t = alpha_1 * ... * alpha_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Précalcul des racines utilisées dans la formule fermée, pour éviter
        # de les recalculer à chaque appel de q_sample.
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Récupère values[t] pour chaque élément du batch, et reshape pour
        pouvoir broadcaster sur une image (B, C, H, W).

        t est un tenseur (B,) d'indices de timestep, potentiellement différent
        pour chaque image du batch (c'est le cas pendant l'entraînement).
        """
        batch_size = t.shape[0]
        out = values.gather(0, t)  # (B,)
        return out.reshape(batch_size, *([1] * (len(shape) - 1)))

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bruite x_0 jusqu'à l'étape t via la formule fermée.

        Args:
            x_0: images propres, shape (B, C, H, W), valeurs dans [-1, 1].
            t: timesteps, shape (B,), valeurs entières dans [0, timesteps).
            noise: bruit à utiliser (sinon généré aléatoirement). Utile pour
                reproduire un bruitage identique lors de tests.

        Returns:
            (x_t, noise) : l'image bruitée et le bruit effectivement utilisé
            (le modèle devra apprendre à prédire ce `noise`).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Reverse process : calcule x_{t-1} à partir de x_t et du bruit prédit par le U-Net.

        Théorie (Ho et al., 2020, éq. 11) : connaissant x_t et une estimation
        du bruit ε_pred qui y a été ajouté, on peut estimer x_{t-1} par :

            x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred)
                      + sigma_t * z,     z ~ N(0, I) si t > 0, sinon z = 0

        avec sigma_t = sqrt(beta_t) (choix standard du papier original DDPM).
        C'est cette fonction, appelée T fois de suite en partant de bruit pur
        (t=T-1 jusqu'à t=0), qui permet de générer une image (voir sample.py).

        Args:
            model_output: bruit prédit par le U-Net, shape (B, C, H, W).
            x_t: image bruitée courante, shape (B, C, H, W).
            t: timesteps courants, shape (B,), identiques pour tout le batch
                en pratique (on débruite tout le batch en parallèle).

        Returns:
            x_{t-1}, shape (B, C, H, W).
        """
        betas_t = self._extract(self.betas, t, x_t.shape)
        alphas_t = self._extract(self.alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        # Moyenne de p(x_{t-1} | x_t), en utilisant le bruit prédit par le modèle.
        mean = (1.0 / torch.sqrt(alphas_t)) * (
            x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * model_output
        )

        # Pas de bruit ajouté à la toute dernière étape (t=0), sinon l'image
        # finale resterait légèrement bruitée.
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        sigma_t = torch.sqrt(betas_t)

        return mean + nonzero_mask * sigma_t * noise


if __name__ == "__main__":
    # Auto-test : vérifie le comportement aux bornes et visualise le bruitage
    # progressif d'une vraie image CelebA.
    import matplotlib.pyplot as plt

    from src.backend.shared.dataset import CelebACaptionDataset

    diffusion = GaussianDiffusion(timesteps=1000)

    # --- Sanity checks numériques ---
    print(f"beta_1={diffusion.betas[0]:.5f}, beta_T={diffusion.betas[-1]:.5f}")
    print(f"alpha_bar_1={diffusion.alphas_cumprod[0]:.5f} (doit être proche de 1)")
    print(f"alpha_bar_T={diffusion.alphas_cumprod[-1]:.5f} (doit être proche de 0)")

    # --- Visualisation sur une vraie image ---
    ds = CelebACaptionDataset(split="train", image_size=64)
    image, caption = ds[0]
    print(f"\nCaption de l'image test : {caption!r}")

    # x_0 et t doivent être sur le même device que les buffers de `diffusion`
    # (mps si dispo) : gather() exige que tous les tenseurs soient colocalisés.
    x_0 = image.unsqueeze(0).to(diffusion.device)  # (1, C, H, W)

    steps_to_show = [0, 50, 100, 250, 500, 750, 999]
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(3 * len(steps_to_show), 3))

    torch.manual_seed(0)
    for ax, t_val in zip(axes, steps_to_show):
        t = torch.tensor([t_val], device=diffusion.device)
        x_t, _ = diffusion.q_sample(x_0, t)

        # Dé-normalise de [-1, 1] vers [0, 1] pour l'affichage (retour CPU pour matplotlib/numpy).
        img_to_show = (x_t[0].clamp(-1, 1) + 1) / 2
        ax.imshow(img_to_show.cpu().permute(1, 2, 0).numpy())
        ax.set_title(f"t={t_val}")
        ax.axis("off")

    plt.tight_layout()
    out_path = "report/forward_diffusion_example.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure sauvegardée dans {out_path}")
