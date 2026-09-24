"""
Training loop pour le DDPM text-to-image.

À chaque pas d'entraînement :
    1. On prend un batch d'images + captions réelles.
    2. On tire un t aléatoire par image du batch.
    3. On bruite les images jusqu'à t (GaussianDiffusion.q_sample) -> x_t, noise.
    4. On encode les captions avec CLIP (gelé) -> text_embed.
    5. Le U-Net prédit le bruit : eps_pred = UNet(x_t, t, text_embed).
    6. loss = MSE(eps_pred, noise) ; backward ; optimizer.step().

Seul le U-Net apprend (CLIP reste gelé, voir shared/text_encoder.py).

Classifier-free guidance (CFG) :
    Pour permettre, au moment de la génération, de renforcer l'influence du
    texte, on entraîne le modèle à fonctionner aussi SANS texte : avec une
    probabilité `cfg_dropout_prob`, on remplace la caption par une chaîne
    vide (le U-Net apprend donc à la fois p(x|texte) et p(x) inconditionnel).

Preview, resume & TensorBoard :
    Toutes les `preview_every_steps` itérations, on génère quelques images
    sur un jeu de captions fixe (pour comparer les checkpoints entre eux au
    fil de l'entraînement), on les sauvegarde dans `<checkpoint_dir>/previews/`
    et on les logge aussi dans TensorBoard (`<checkpoint_dir>/tensorboard/`).
    Un entraînement interrompu peut être repris via `resume=<checkpoint.pt>`.

Ce module n'a pas de CLI propre : il est appelé par src/cli.py (`python -m
src.cli train diffusion ...`), qui centralise l'argument parsing pour les
deux approches (diffusion et flow_matching).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.backend.diffusion.gaussian_diffusion import GaussianDiffusion
from src.backend.diffusion.sample import generate as sample_ddpm
from src.backend.shared.dataset import CelebACaptionDataset
from src.backend.shared.device import get_device
from src.backend.shared.preview import save_preview_grid
from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
from src.backend.shared.unet import UNet

# Captions fixes utilisées pour les previews, afin de pouvoir comparer
# visuellement la progression d'un checkpoint à l'autre.
PREVIEW_CAPTIONS = [
    "a young woman smiling with bangs",
    "a man with a beard and no smile",
    "an elderly person wearing eyeglasses",
    "a smiling young man with no beard",
]


def train(
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-4,
    image_size: int = 64,
    timesteps: int = 1000,
    cfg_dropout_prob: float = 0.1,
    limit_batches: int | None = None,
    checkpoint_dir: str = "checkpoints/diffusion",
    checkpoint_every_steps: int = 500,
    preview_every_steps: int = 500,
    log_every_steps: int = 50,
    resume: str | None = None,
    device: str | torch.device | None = None,
) -> None:
    """Entraîne le U-Net du DDPM sur CelebA-Dialog.

    Args:
        epochs: nombre de passages complets sur le dataset.
        batch_size: taille de batch.
        lr: learning rate (Adam).
        image_size: résolution (carrée) des images.
        timesteps: nombre d'étapes T du schedule de diffusion.
        cfg_dropout_prob: probabilité de remplacer la caption par une chaîne
            vide pendant l'entraînement (classifier-free guidance).
        limit_batches: si fourni, arrête chaque epoch après ce nombre de
            batches (utile pour un smoke test rapide avant un run complet).
        checkpoint_dir: dossier où sauvegarder les poids du U-Net, les
            previews (checkpoint_dir/previews/) et les logs TensorBoard
            (checkpoint_dir/tensorboard/).
        checkpoint_every_steps: fréquence de sauvegarde (en pas d'optimisation).
        preview_every_steps: fréquence de génération d'images de contrôle.
        log_every_steps: fréquence d'affichage/logging de la loss.
        resume: chemin vers un checkpoint .pt à partir duquel reprendre
            l'entraînement (poids du U-Net + optimizer + step). None pour
            démarrer de zéro.
        device: device torch (mps par défaut sur ce projet).
    """
    device = torch.device(device) if device is not None else get_device()
    print(f"Device: {device}")

    dataset = CelebACaptionDataset(split="train", image_size=image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")

    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    text_encoder = FrozenCLIPTextEncoder(device=device)
    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"U-Net: {n_params:,} paramètres entraînables")

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    preview_path = checkpoint_path / "previews"

    writer = SummaryWriter(log_dir=str(checkpoint_path / "tensorboard"))
    writer.add_text(
        "config",
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, image_size={image_size}, "
        f"timesteps={timesteps}, cfg_dropout_prob={cfg_dropout_prob}, params={n_params:,}",
    )

    global_step = 0
    start_epoch = 0
    if resume is not None:
        global_step, start_epoch = _load_checkpoint(unet, optimizer, resume)
        print(f"Reprise depuis {resume} : step={global_step}, epoch={start_epoch + 1}")

    # Embeddings des captions de preview, calculés une seule fois (CLIP est gelé).
    preview_text_embed = text_encoder(PREVIEW_CAPTIONS)

    unet.train()

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            running_loss = 0.0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, (images, captions) in enumerate(progress):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break

                images = images.to(device)

                # Classifier-free guidance : on "efface" le texte pour une fraction
                # du batch, pour que le modèle apprenne aussi le cas inconditionnel.
                captions = [
                    "" if torch.rand(1).item() < cfg_dropout_prob else c for c in captions
                ]

                t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)
                x_t, noise = diffusion.q_sample(images, t)

                text_embed = text_encoder(captions)
                eps_pred = unet(x_t, t, text_embed)

                loss = torch.nn.functional.mse_loss(eps_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % log_every_steps == 0:
                    avg_loss = running_loss / log_every_steps
                    progress.set_postfix(loss=f"{avg_loss:.4f}")
                    writer.add_scalar("loss/train", avg_loss, global_step)
                    running_loss = 0.0

                if global_step % checkpoint_every_steps == 0:
                    _save_checkpoint(unet, optimizer, global_step, epoch, checkpoint_path / "latest.pt")

                if global_step % preview_every_steps == 0:
                    _run_preview(
                        unet, diffusion, preview_text_embed, preview_path, global_step, device, writer
                    )
                    unet.train()  # _run_preview met le modèle en eval()

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{epochs} terminée en {epoch_time / 60:.1f} min")
            writer.add_scalar("time/epoch_minutes", epoch_time / 60, epoch)
            _save_checkpoint(unet, optimizer, global_step, epoch, checkpoint_path / f"epoch_{epoch + 1}.pt")

        print("Entraînement terminé.")
    finally:
        writer.close()


@torch.no_grad()
def _run_preview(
    unet: torch.nn.Module,
    diffusion: GaussianDiffusion,
    text_embed: torch.Tensor,
    preview_path: Path,
    step: int,
    device: torch.device,
    writer: SummaryWriter,
) -> None:
    images = sample_ddpm(unet, diffusion, text_embed, device=device, show_progress=False)
    save_preview_grid(
        images,
        PREVIEW_CAPTIONS,
        preview_path / f"step_{step:07d}.png",
        title=f"DDPM — step {step}",
    )
    # Grille d'images normalisées [0, 1] pour TensorBoard (add_images attend NCHW dans [0,1]).
    images_01 = (images.clamp(-1, 1) + 1) / 2
    writer.add_images("preview/samples", images_01, global_step=step)


def _save_checkpoint(
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(
    unet: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str
) -> tuple[int, int]:
    """Charge un checkpoint et retourne (global_step, epoch_à_reprendre).

    L'epoch sauvegardée est celle qui vient d'être complétée (ou en cours
    pour `latest.pt`) ; on reprend à la même epoch pour rester simple (les
    quelques batches déjà vus dans cette epoch seront revus, ce qui est sans
    conséquence pour ce genre d'entraînement).
    """
    checkpoint = torch.load(path, map_location=next(unet.parameters()).device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint.get("epoch", 0)
