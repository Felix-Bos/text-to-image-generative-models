"""
Flow Matching (Conditional Flow Matching, Lipman et al., 2023) : approche
alternative au DDPM pour entraîner un modèle génératif, en réutilisant le
même U-Net (voir shared/unet.py) que la diffusion.

Différence de principe avec le DDPM (gaussian_diffusion.py) :

    DDPM  : bruitage par petits pas discrets t=0..T-1, schedule beta_t appris
            empiriquement, le modèle prédit le BRUIT ajouté.

    Flow Matching : interpolation LINÉAIRE et continue entre l'image x_0 et
            un bruit gaussien x_1, pour t continu dans [0, 1] :

                x_t = (1 - t) * x_0 + t * noise

            Le modèle apprend à prédire le CHAMP DE VITESSE constant le long
            de ce segment de droite :

                v = noise - x_0        (dérivée de x_t par rapport à t)

            La loss est simplement MSE(v_pred, v).

Intuition : au lieu d'apprendre à "retirer du bruit petit à petit" comme le
DDPM, le modèle apprend directement "dans quelle direction et à quelle
vitesse se déplacer" pour aller d'un point bruité vers l'image cible. À la
génération, on résout l'équation différentielle dx/dt = v_pred(x, t) à
rebours (de t=1 vers t=0) avec un solveur ODE simple (voir sample.py).

Avantage pratique attendu : le chemin étant une droite (pas une marche
aléatoire), on peut généralement utiliser beaucoup moins d'étapes de
génération qu'un DDPM classique pour un résultat comparable.
"""

from __future__ import annotations

import torch


class ConditionalFlowMatching:
    """Gère l'interpolation entre image et bruit, et le champ de vitesse cible.

    Contrairement à GaussianDiffusion, il n'y a pas de schedule beta_t à
    précalculer : t est un scalaire continu dans [0, 1], tiré uniformément
    pendant l'entraînement.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        from src.backend.shared.device import get_device

        self.device = torch.device(device) if device is not None else get_device()

    def sample_xt(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpole entre l'image x_0 et un bruit gaussien, à l'instant t.

        Args:
            x_0: images propres, shape (B, C, H, W), valeurs dans [-1, 1].
            t: instants, shape (B,), valeurs continues dans [0, 1].
            noise: bruit à utiliser (sinon généré aléatoirement).

        Returns:
            (x_t, noise, velocity) :
                x_t      : point interpolé, shape (B, C, H, W).
                noise    : le bruit effectivement utilisé (x_1).
                velocity : la cible que le modèle doit apprendre à prédire,
                           velocity = noise - x_0 (constante le long du chemin).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Reshape t pour broadcaster sur (B, C, H, W).
        t_reshaped = t.reshape(-1, *([1] * (len(x_0.shape) - 1)))

        x_t = (1.0 - t_reshaped) * x_0 + t_reshaped * noise
        velocity = noise - x_0

        return x_t, noise, velocity


if __name__ == "__main__":
    # Auto-test : vérifie le comportement aux bornes (t=0 -> x_0, t=1 -> noise)
    # et visualise l'interpolation progressive d'une vraie image CelebA.
    import matplotlib.pyplot as plt

    from src.backend.shared.dataset import CelebACaptionDataset

    flow = ConditionalFlowMatching()
    print(f"Device: {flow.device}")

    ds = CelebACaptionDataset(split="train", image_size=64)
    image, caption = ds[0]
    print(f"\nCaption de l'image test : {caption!r}")

    x_0 = image.unsqueeze(0).to(flow.device)

    torch.manual_seed(0)
    fixed_noise = torch.randn_like(x_0)

    # --- Sanity checks aux bornes ---
    x_t0, _, v0 = flow.sample_xt(x_0, torch.tensor([0.0], device=flow.device), noise=fixed_noise)
    x_t1, _, v1 = flow.sample_xt(x_0, torch.tensor([1.0], device=flow.device), noise=fixed_noise)

    err_t0 = (x_t0 - x_0).abs().max().item()
    err_t1 = (x_t1 - fixed_noise).abs().max().item()
    print(f"\nÀ t=0, x_t doit être égal à x_0. Erreur max: {err_t0:.6f} (doit être ~0)")
    print(f"À t=1, x_t doit être égal au bruit. Erreur max: {err_t1:.6f} (doit être ~0)")

    # La vélocité doit être constante le long du chemin (indépendante de t).
    v_diff = (v0 - v1).abs().max().item()
    print(f"La vélocité doit être identique à tout t. Différence: {v_diff:.6f} (doit être ~0)")

    # --- Visualisation de l'interpolation progressive ---
    steps_to_show = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(3 * len(steps_to_show), 3))

    for ax, t_val in zip(axes, steps_to_show):
        t = torch.tensor([t_val], device=flow.device)
        x_t, _, _ = flow.sample_xt(x_0, t, noise=fixed_noise)

        img_to_show = (x_t[0].clamp(-1, 1) + 1) / 2
        ax.imshow(img_to_show.cpu().permute(1, 2, 0).numpy())
        ax.set_title(f"t={t_val:.2f}")
        ax.axis("off")

    plt.tight_layout()
    out_path = "report/flow_matching_interpolation_example.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure sauvegardée dans {out_path}")
